# Credits: Foodiemonster007

# About

This tool helps you convert text from images into a digital format, especially for screenshots of book chapters. Here's a simplified breakdown of how to use it.

# Requirements

1) Python: This is a programming language that makes the tool work. You'll need to install it from the official website (https://www.python.org/downloads/). Make sure to check the 2 boxes that say "Add Python to PATH" and "pip" during installation. This step lets your computer easily find and run the program.

2) Fonts: To get the best results, the tool needs specific fonts: Arial and Malgun/Microsoft Yahei/MS Gothic. If your Windows computer is set up for both English and Korean/Chinese/Japanese languages, you probably already have them.

3) Google API Key: This is a special password that connects the program to Google's text-reading service. You can get a free one from the Google Cloud Console (https://console.cloud.google.com/). Here are the instructions on how to do it: https://www.youtube.com/watch?v=brCkpzAD0gc

4) For Mac users: The program file has a special ending. If you're on a Mac, you'll need to change the file's name by replacing .pyw with .py

# Instructions

1.  Place a folder with your images inside the "screenshots" folder. I have provided a folder named "sample" as an example.

2.  Start the Program: Find the program file (which should now be named `novelOCR.pyw` or `novelOCR.py` on a Mac) and double-click it.

3.  Enter Your API Key: When the program opens, it will ask for the Google API key you got earlier. This is a crucial step; the tool won't work without it.

4. Edit the settings according to the instructions below, and hit Run Process.

You can also improve the quality of the text conversion by adjusting the settings in the program:

- Adjust the AI Prompt: Think of this as giving the AI instructions. You can tell it what to look for, like Hanja (Chinese characters used in Korean), English letters, or emoticons.
		Example AI Prompts:
		Korean: Perform OCR on this image. The text is in Korean and may include Hanja characters and repetitive sound effects (e.g., '콰콰콰콰콰' or '스스스스스'). Transcribe the text exactly as it appears, preserving all original characters including Hanja.
		Chinese: Perform OCR on this image. The text is in Chinese. Transcribe the main text exactly as it appears.
		Japanese: Perform OCR on this image. The text is in Japanese and has some furigana. Transcribe the main text exactly as it appears, with the furigana in brackets next to the kanji.

- Choose the Model: The tool uses different versions of Google's Gemini AI to convert text. You can try different ones to see which works faster or gives better quality results.

- Chapter Title Detection: The program looks for a specific pattern to identify chapter titles. The default is something like 제0회, where the 0 can be any number. If your chapters are titled differently (e.g. 1화, 2화, 3화, etc.) you'll need to update the pattern to ^\s*\d+화. *If you still haven't got a clue what this is then leave it untouched and let this part fail, it won't affect the ocr significantly.*

- OCR Image Enhancement: Try checking/unchecking the Unsharp Mask and Binarization checkboxes to see if the OCR quality improves. They don't do much for screenshots but are highly effective at enhancing the clarity of photos.

- Image Replacements: Sometimes, novels use unusual symbols and East Asian punctuation for things like ellipses (the "⋯" symbol) or separators ("∗∗∗" or "☆☆☆"). You can train the program to replace these symbols with a specific character, like @ for an ellipsis. *If you still haven't got a clue what this is then "Delete All Replacements" and let this part fail, it won't affect the ocr significantly.*
	1. You'll need to create small, perfect image files of these symbols from your screenshot to teach the program what to look for. 
	2. You should adjust the font size to make sure the replacement text for OCR (@) takes up the same amount of space as the original image.
	3. You can also change the threshold, which is a number between 0.5 and 1, to tell the program how close of a match the image needs to be before a replacement is performed.
	4. Type in the desired text you want to see in the OCR'd document, e.g. "..." for ellipsis or "***" for separators. The previous placeholder text (e.g. @) will be changed to your desired text.

- SAVE CONFIG FOR THE PROGRAM TO REMEMBER YOUR CUSTOM SETTINGS!

# Frequently Asked Questions

1. If pip is missing: Uninstall and reinstall python, remember to check the boxes for "pip" and "Add Python to PATH"

2. Packages should install automatically the first time you open the .pyw file. If packages refuse to install automatically because of admin permissions or something: Open command prompt and key in: pip install opencv-python numpy pillow google-generativeai

3. If your merged images are out of order: The script should merge images in alphabetical/numerical order, but sometimes it hates numbers without leading zeroes. When it sees 1.jpg, 2.jpg, it merges in the order 1.jpg, 10,jpg, 11.jpg instead. Rename the image files by adding zeroes e.g. 01.jpg and 02.jpg to correct the merging order.