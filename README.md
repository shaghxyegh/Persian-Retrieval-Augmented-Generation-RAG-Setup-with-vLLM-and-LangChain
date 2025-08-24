# Persian-Retrieval-Augmented-Generation-RAG-Setup-with-vLLM-and-LangChain
Follow-up RAG for Persian QA on Colab T4 GPU with vLLM and TitanML/ChatQA-8B-AWQ. Overcame PDF loading errors, VRAM limits, and Transformers complexities. Set up LangChain for embeddings, ChromaDB, and Persian PDFs. ~43 toks/s input, ~24 toks/s output. 
# Project Report: Persian Retrieval-Augmented Generation (RAG) Setup with vLLM and LangChain

## Project Overview
This project builds on my previous experience with Retrieval-Augmented Generation (RAG) to create a system tailored for Persian language question-answering. Using Google Colab’s free T4 GPU, I set up the TitanML/ChatQA-1.5-8B-AWQ-4bit model with vLLM for efficient inference, integrated with LangChain for Persian document retrieval. The report is written in English, unlike the Persian focus of the data processing.

**Objectives**:
- Install LangChain, vLLM, and RAG dependencies for Persian text processing.
- Load a quantized LLM (TitanML/ChatQA-1.5-8B-AWQ-4bit) for memory-efficient inference.
- Prepare embeddings, vector storage (Chroma), and PDF parsing for Persian documents.
- Test model inference on Persian prompts to ensure functionality.

**Tech Stack**:
- Python libraries: LangChain (langchain_huggingface, langchain_chroma), vLLM, Sentence Transformers, ChromaDB, PyMuPDF.
- Model: TitanML/ChatQA-1.5-8B-AWQ-4bit (quantized, FP16).
- Environment: Google Colab (free T4 GPU, ~15GB VRAM).

## Methodology
1. **Dependency Installation**:
   - Installed/upgraded LangChain components (langchain_core, langchain_community) and vLLM for efficient LLM serving.
   - Added RAG libraries: sentence_transformers for embeddings, chromadb for vector storage, and pymupdf for Persian PDF parsing.

2. **LLM Initialization**:
   - Loaded the AWQ-quantized ChatQA model with vLLM, using FP16 data type and a max sequence length of 8192.
   - Configured for eager mode on CUDA (T4 GPU) to manage memory constraints.

3. **Testing**:
   - Tested model with a sample Persian prompt to verify inference performance (e.g., token speed, coherence).
   - Used vLLM logs to monitor initialization and generation metrics.

## Results
- Successfully installed dependencies and loaded the quantized model on Colab’s T4 GPU.
- Test generation: Produced coherent output for a Persian prompt (e.g., greeting response) with ~43 input tokens/s and ~24 output tokens/s.
- Setup is ready for Persian RAG: document ingestion, embedding, vector storage, and retrieval-augmented QA.

## Challenges and Solutions
1. **Document Loading Issues**:
   - **Challenge**: Failed to load Persian PDFs directly due to access restrictions or format errors, similar to previous JSON loading issues (`RuntimeError: Dataset scripts are no longer supported`).
   - **Solution**: Manually uploaded Persian PDFs to Colab and used PyMuPDF for parsing. Ensured proper encoding for Persian text.

2. **Colab T4 GPU Limitations**:
   - **Challenge**: Limited VRAM (~15GB) and session timeouts risked out-of-memory errors for the 8B model.
   - **Solution**: Used AWQ 4-bit quantization to reduce memory usage. Enforced vLLM’s eager mode and monitored VRAM via logs. Future: Use Colab Pro for better resources.

3. **Transformers Library Complexity**:
   - **Challenge**: Latest Transformers version (4.55.3) and vLLM/LangChain integrations had complexities, e.g., renamed `eval_strategy` and compatibility issues.
   - **Solution**: Ignored `trust_remote_code` warnings, adapted to new parameter names, and pinned versions (e.g., `evaluate==0.4.5`). Consulted vLLM/Hugging Face docs.

4. **Persian Text Processing**:
   - **Challenge**: Handling right-to-left (RTL) Persian text and ensuring embeddings capture semantic nuances.
   - **Solution**: Used Sentence Transformers compatible with multilingual text; tested with Persian prompts to verify coherence. Future: Fine-tune embeddings for Persian.

## Future Work
- **Full RAG Pipeline**: Implement Persian document chunking, embedding, and querying with ChromaDB.
- **Enhanced Persian Support**: Use Persian-specific embeddings (e.g., ParsBERT-based) for better retrieval.
- **Performance Optimization**: Tune vLLM batch sizes and embedding parameters.
- **Deployment**: Build a FastAPI or Hugging Face API for Persian QA.
- **Environment Upgrade**: Use Colab Pro or local GPU for stability.

## References
- vLLM Documentation: https://docs.vllm.ai
- LangChain Documentation: https://python.langchain.com/docs
- Model: https://huggingface.co/TitanML/ChatQA-1.5-8B-AWQ-4bit
- Hugging Face Transformers: https://huggingface.co/docs/transformers

**Author**: Shaghayegh Shafiee  
**Date**: August 24, 2025
