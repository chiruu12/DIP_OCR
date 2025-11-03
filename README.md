# Custom OCR System for Book Digitization

## 1. Project Overview

This project implements a complete, high-performance Optical Character Recognition (OCR) system built from scratch in PyTorch. The primary goal is to accurately digitize text from scanned PDF documents.

The project documents a full engineering journey, starting with a classic but flawed architecture and culminating in the implementation of a modern, state-of-the-art CRNN (Convolutional Recurrent Neural Network) model. The final model is trained exclusively on a "ground truth" dataset generated directly from the source PDFs, enabling it to achieve high accuracy on the specific fonts and noise profiles of the documents.

## 2. The Engineering Journey: From Failure to Success

### Initial Approach: "Segment-then-Recognize" (Failure)

Our first approach was a classic OCR pipeline with a **Triage + Expert Model** architecture:
1.  **Segmentation:** Use OpenCV (`cv2.findContours`) to draw a bounding box around every individual character on the page.
2.  **Triage Model:** A CNN designed to classify each character box as a `digit`, `uppercase`, or `lowercase` letter.
3.  **Expert Models:** Three separate CNNs, each specialized in recognizing characters within its assigned class.

**Why it Failed:** This architecture proved to be fundamentally flawed for real-world scanned documents.
*   **Segmentation Failure:** The system was unable to correctly segment characters that were touching due to font kerning, ligatures (`fi`, `fl`), or scanner noise. It often identified whole words or multiple characters as a single, unrecognizable "blob."
*   **Data Mismatch:** The models, trained on perfectly isolated characters, produced garbage output when fed these malformed, multi-character blobs.

The result was incoherent, unusable text, proving that a pre-segmentation step is too fragile for this task.

### Final Approach: CRNN (Success)

We pivoted to the industry-standard architecture for OCR: a **Convolutional Recurrent Neural Network (CRNN)**. This approach solves the fundamental flaws of the previous method.

**How it Works:**
1.  **Input:** The model processes an **entire line of text** as a single image, completely bypassing the need for fragile single-character segmentation.
2.  **CNN Backbone (The "Eyes"):** A deep convolutional network scans the line image from left to right, extracting a sequence of rich feature vectors.
3.  **RNN Processor (The "Brain"):** A bi-directional LSTM network reads this sequence of features, using the order and context to understand how features form characters and words. This is how it naturally handles touching and connected letters.
4.  **CTC Loss (The "Translator"):** The model is trained with a Connectionist Temporal Classification (CTC) loss function. This powerful algorithm allows the model to learn how to align its sequence of predictions with the ground-truth text label, without needing to be told where each character is.

## 3. The Ground-Truth Dataset

To ensure the highest accuracy, we abandoned purely synthetic data. The `create_real_dataset.py` script implements a "ground truth" pipeline:
1.  **Rich Text Extraction:** It uses `PyMuPDF` to extract every word from the source PDFs along with its precise `(x, y)` coordinates on the page.
2.  **Line Image Extraction:** It uses OpenCV to find the bounding boxes of text lines on the scanned page image.
3.  **Automatic Alignment:** It matches the words-with-coordinates to the line-image-boxes, automatically generating a perfectly labeled ground-truth pair of `(real_line_image, "correct_line_text")`.
4.  **HDF5 Storage:** This final, high-quality dataset is stored in a single, efficient `real_line_dataset.h5` file for fast training.

## 4. Model Training

The `train_real_data.py` script trains the CRNN model from scratch on our custom ground-truth dataset.

**Training Details:**
-   **Architecture:** Deep CRNN with Batch Normalization.
-   **Loss Function:** `nn.CTCLoss`.
-   **Optimizer:** Adam.
-   **Scheduler:** `StepLR` to manage the learning rate.
-   **Validation:** After each epoch, the script performs a full OCR on a real PDF page to provide a true, real-world benchmark of the model's progress.

## 5. How to Use the Project

### a. Setup
Install the required libraries:
```bash
pip install -r requirements.txt
```

### b. Step 1: Create the Dataset
Place your source PDF files in the `sample_documents/books/` directory. Then, run the data creation script. This only needs to be done once.
```bash
python create_real_dataset.py --clean
```

### c. Step 2: Train the Model
Run the training script. This will process the `real_line_dataset.h5` file and save the final trained model to `models/crnn_final/`.
```bash
python train_real_data.py
```

### d. Step 3: Run OCR
Use the final application script to perform OCR on any page of a PDF.

Example:
```bash
python run_crnn_ocr.py "sample_documents/books/Applied-Machine-Learning-and-AI-for-Engineers.pdf" --page 2
```
