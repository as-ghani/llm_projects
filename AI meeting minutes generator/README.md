# AI Meeting Minutes Generator

An end-to-end AI meeting minutes generator that takes a recorded meeting, converts the audio into a text transcript, and uses a large language model to generate structured meeting minutes.

The project demonstrates a pipeline combining **speech-to-text** and **large language models (LLMs)** to automate the process of turning a meeting recording into useful, structured notes.

## Overview

The application follows this pipeline:

```text
Meeting Audio
     ↓
Speech-to-Text
     ↓
Meeting Transcript
     ↓
Llama 3.2 3B Instruct
     ↓
Structured Meeting Minutes
```

For speech recognition, the notebook demonstrates open source approache:

1. **Open-source Whisper Medium** running through Hugging Face Transformers.

The resulting transcript is then passed to **Meta's Llama 3.2 3B Instruct** model, which generates structured meeting minutes.

## Features

- Convert meeting audio into text
- Use Hugging Face's Whisper model for open-source transcription
- Generate structured meeting minutes using Llama 3.2
- Extract discussion points and key takeaways
- Generate action items and identify their owners
- Use 4-bit quantization to reduce GPU memory requirements
- Designed to run in Google Colab with a CUDA-enabled GPU



## Generated Meeting Minutes

The language model is instructed to generate minutes containing:

- **Summary**
- **Attendees**
- **Location**
- **Date**
- **Discussion points**
- **Key takeaways**
- **Action items**
- **Action-item owners**



## Technologies Used


| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| Python                    | Main programming language       |
| Google Colab              | Development and GPU environment |
| PyTorch                   | Model inference                 |
| Hugging Face Transformers | Model loading and inference     |
| Whisper Medium            | Speech-to-text transcription    |
| Llama 3.2 3B Instruct     | Meeting-minutes generation      |
| BitsAndBytes              | 4-bit model quantization        |




## Models



### Whisper Medium

The project uses:

`openai/whisper-medium.en`

Whisper converts the meeting recording into a text transcript that can then be processed by the language model.

### Llama 3.2 3B Instruct

The project uses:

`meta-llama/Llama-3.2-3B-Instruct`

Llama receives the transcript together with a structured prompt and generates the final meeting minutes.

The Llama model is loaded using 4-bit quantization to reduce GPU memory usage.

## Running the Project

The notebook was developed and executed using **Google Colab with a GPU runtime**.

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd llm-projects/AI\ meeting\ minutes\ generator
```



### 2. Open the notebook

Open:

```text
generator.ipynb
```

in Google Colab.

### 3. Add the audio file

The notebook expects the meeting recording to be available at:

```text
Content/
     └── denver_extract.mp3
```

If you use a different audio file, update the `audio_filename` variable in the notebook.

### 4. Configure Hugging Face

The Llama model is hosted on Hugging Face.

Add your Hugging Face token to Google Colab Secrets as:

```text
HF_TOKEN
```

The notebook retrieves the token securely using:

```python
userdata.get("HF_TOKEN")
```

You should **never place your actual Hugging Face token directly inside the notebook or commit it to GitHub**.


## Google Colab

The easiest way to reproduce the project is to run the notebook in Google Colab with a GPU runtime.

The Llama inference portion uses CUDA-specific 4-bit quantization through BitsAndBytes, so the notebook is primarily configured for an NVIDIA CUDA environment.

## Running Locally

The models can also be run locally on sufficiently powerful hardware.

However, the notebook's current configuration is designed around **NVIDIA CUDA GPUs** and Google Colab. In particular, the BitsAndBytes 4-bit configuration used in the notebook should not be assumed to work unchanged on Apple Silicon.

The notebook is therefore kept in its original Colab-oriented form so that the project can be reproduced using the same environment in which it was developed.

## Project Structure

```text
AI meeting minutes generator/
├── denver_extract.mp3
├── generator.ipynb
└── README.md
```



### `generator.ipynb`

The main notebook containing the complete pipeline:

1. Install dependencies
2. Mount Google Drive
3. Authenticate with Hugging Face
4. Load the meeting audio
5. Transcribe the audio using Whisper
6. Construct the meeting-minutes prompt
7. Load Llama 3.2 3B
8. Generate the meeting minutes
9. Display the final result



### `denver_extract.mp3`

The meeting recording used as the input to the transcription pipeline.

## Example Output

The generated output is structured approximately as follows:

```markdown
# Meeting Minutes

## Summary

A summary of the meeting...

## Attendees

- Person 1
- Person 2
- Person 3

## Discussion Points

- Topic discussed...
- Topic discussed...

## Key Takeaways

- Important takeaway...
- Important takeaway...

## Action Items

| Action Item | Owner |
|---|---|
| Follow up on... | Person 1 |
| Review... | Person 2 |
```

The exact output depends on the contents of the meeting transcript and the model's generation.

## What I Learned

This project explores several important concepts in modern AI application development:

- Running open-source AI models with Hugging Face Transformers
- Using Whisper for automatic speech recognition
- Prompting an instruction-tuned LLM
- Using quantization to reduce model memory requirements
- Running LLM inference on GPU hardware
- Building a multi-stage AI pipeline
- Managing API keys and authentication securely
- Working with Google Colab and Google Drive for AI projects



## Notes

The notebook contains an open-source Whisper transcription approach.

The Llama model requires access through Hugging Face and may require accepting the model's applicable terms before it can be downloaded.

## License

This project is for educational and demonstration purposes.