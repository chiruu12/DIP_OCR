# DIP_OCR




```
ocr_project/
|
|
|-- models/
|   |-- digits_model.pth      (Our trained 'Small' model for digits)
|   |-- uppercase_model.pth   (Our trained 'Medium' model for uppercase)
|   |-- lowercase_model.pth   (Our trained 'Medium' model for lowercase)
|
|-- notebooks/
|   |-- Model_Training_and_Comparison.ipynb  (The Kaggle notebook)
|
|-- src/
|   |-- __init__.py
|   |-- utils.py          (Will contain image processing functions)
|   |-- model_loader.py   (Will contain logic to load our models)
|   |-- ocr_pipeline.py   (Will contain the main OCR logic)
|   |-- main.py           (The main script to run the application)
|
|-- sample_documents/
|   |-- o_rielly_nn_page.png (A sample image to test our OCR on)
|
|-- requirements.txt      (Python dependencies)
|-- README.md             (Project description and instructions)
|-- RESULTS.md            (The results file we just created)
```