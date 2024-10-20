 Structured Text Extraction from Images using OCR and LLM

This project demonstrates how to extract structured text (headings and subheadings) from images using OCR (Optical Character Recognition) with Tesseract and further organize it using a Groq-powered Language Model (LLM). The result is a clean, organized JSON structure with headings as keys and subheadings or bullet points as values.



 Features
- Preprocessing with OpenCV: Converts input images to grayscale and applies thresholding to enhance text visibility.
- OCR with Tesseract: Extracts text from the processed image.
- Organization with LLM: Uses an LLM to analyze and convert the extracted text into a meaningful JSON structure.
- Error Handling: Includes fallback for raw model responses in case of JSON parsing errors.



 Dependencies

1. Python 3.x  
2. Tesseract OCR: Install via package manager:
   bash
   sudo apt update
   sudo apt install tesseract-ocr
   
3. Libraries: Install required Python libraries:
   bash
   pip install opencv-python pytesseract langchain_groq
   



 Usage Instructions

1. Clone or Download the Repository.

2. Set the Path for Tesseract:  
   In the code, ensure that the path to the Tesseract executable is correct:
   python
   pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
   

3. Prepare an Image:  
   Use an image that contains headings and subheadings (e.g., `"India.png"`).

4. Run the Notebook or Script:
   If working with the provided notebook, open it and run the cells in sequence.  
   Alternatively, run the script from the command line:
   bash
   python your_script_name.py
   



 Approach and Methods

 1. Image Preprocessing with OpenCV
- Grayscale Conversion: Simplifies the image by reducing it to a single color channel.
- Binary Thresholding: Converts the grayscale image to black and white to enhance text clarity.

   python
   image = cv2.imread(image_path)
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
   

 2. Text Extraction using Tesseract
The preprocessed binary image is passed to Tesseract OCR to extract text:
   python
   ocr_data = pytesseract.image_to_string(binary_image, config='--oem 3 --psm 6')
   

 3. Querying the LLM
The extracted text is sent to a Groq-powered language model with a well-defined prompt to generate structured output:
   python
   prompt = f"""
   Analyze the text and convert it into a JSON format where:
   - Keys are headings.
   - Values are lists of points or subheadings under each heading.
   Text: {text}
   """
   response = llm.invoke(prompt).content
   

 4. Error Handling and Fallbacks
If the JSON parsing fails, the raw response from the model is printed to aid in debugging.



 Code Walkthrough

1. Preprocessing Function:
   python
   def preprocess_image(image_path):
       image = cv2.imread(image_path)
       gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
       ocr_data = pytesseract.image_to_string(binary_image, config='--oem 3 --psm 6')
       return ocr_data.strip()
   

2. LLM Query Function:
   python
   def query_groq_model(text):
       llm = ChatGroq(
           temperature=0,
           groq_api_key='your_groq_api_key',
           model_name="llama-3.1-70b-versatile"
       )
       prompt = f"Convert this text into JSON with headings and subheadings: {text}"
       response = llm.invoke(prompt).content
       try:
           return json.loads(response)
       except json.JSONDecodeError:
           return response
   

3. Main Function:
   python
   if __name__ == "__main__":
       image_path = "India.png"
       extracted_text = preprocess_image(image_path)
       output_data = query_groq_model(extracted_text)
       print("Final Output:", json.dumps(output_data, indent=4))
   



 Example Output

Given an input image with the following structure:

National Symbols
- Flag
- Emblem

States
- Maharashtra
- Gujarat


The output will be:
json
{
    "National Symbols": ["Flag", "Emblem"],
    "States": ["Maharashtra", "Gujarat"]
}




 Troubleshooting

- OCR Accuracy Issues:  
  - Ensure the image has high contrast text.
  - Adjust the thresholding value if necessary.

- LLM Response Not in JSON:  
  - Check for syntax errors in the prompt.
  - Print the raw response for debugging if JSON parsing fails.



 Conclusion

This project leverages computer vision and natural language processing to extract meaningful information from images and present it in a structured format. It’s particularly useful for automating text extraction tasks from scanned documents, forms, and other visual data.
