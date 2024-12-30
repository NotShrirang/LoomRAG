# 🌟 LoomRAG: Multimodal Retrieval-Augmented Generation for AI-Powered Search

![GitHub stars](https://img.shields.io/github/stars/NotShrirang/LoomRAG?style=social)
![GitHub forks](https://img.shields.io/github/forks/NotShrirang/LoomRAG?style=social)
![GitHub commits](https://img.shields.io/github/commit-activity/t/NotShrirang/LoomRAG)
![GitHub issues](https://img.shields.io/github/issues/NotShrirang/LoomRAG)
![GitHub pull requests](https://img.shields.io/github/issues-pr/NotShrirang/LoomRAG)
![GitHub](https://img.shields.io/github/license/NotShrirang/LoomRAG)
![GitHub last commit](https://img.shields.io/github/last-commit/NotShrirang/LoomRAG)
![GitHub repo size](https://img.shields.io/github/repo-size/NotShrirang/LoomRAG)
![Streamlit Playground](https://img.shields.io/badge/Streamlit%20App-red?style=flat-rounded-square&logo=streamlit&labelColor=white)

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system, named **LoomRAG**, that leverages OpenAI's CLIP model for neural cross-modal retrieval and semantic search. The system allows users to input text queries and retrieve both text and image responses seamlessly through vector embeddings. It also supports uploading images and PDFs for enhanced interaction and intelligent retrieval capabilities through a Streamlit-based interface.

---
## 📸 Implementation Screenshots

| ![Screenshot 2024-12-30 111906](https://github.com/user-attachments/assets/13c0bd0d-1569-4d9e-aae5-ea5801a69beb) | ![Screenshot 2024-12-30 114200](https://github.com/user-attachments/assets/d74e9d75-7716-4705-9564-0c6fdc26790b) |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Screenshot 1                                                                                                  | Screenshot 2                                                                                                  |

---

## ✨ Features
- 🔄 **Cross-Modal Retrieval**: Search text to retrieve both text and image results using deep learning
- 🌐 **Streamlit Interface**: Provides a user-friendly web interface for interacting with the system
- 📤 **Upload Options**: Allows users to upload images and PDFs for AI-powered processing and retrieval
- 🧠 **Embedding-Based Search**: Uses OpenAI's CLIP model to align text and image embeddings in a shared latent space
- 🔍 **Augmented Text Generation**: Enhances text results using LLMs for contextually rich outputs

---

## 🏗️ Architecture Overview
1. **Data Indexing**:
   - Text, images, and PDFs are preprocessed and embedded using the CLIP model
   - Embeddings are stored in a vector database for fast and efficient retrieval

2. **Query Processing**:
   - Text queries are converted into embeddings for semantic search
   - Uploaded images and PDFs are processed and embedded for comparison
   - The system performs a nearest neighbor search in the vector database to retrieve relevant text and images

3. **Response Generation**:
   - For text results: Optionally refined or augmented using a language model
   - For image results: Directly returned or enhanced with image captions
   - For PDFs: Extracts text content and provides relevant sections

---

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NotShrirang/LoomRAG.git
   cd LoomRAG
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage
1. **Running the Streamlit Interface**:
   - Start the Streamlit app:

     ```bash
     streamlit run app.py
     ```
   - Access the interface in your browser to:
     - Submit natural language queries
     - Upload images or PDFs to retrieve contextually relevant results

2. **Example Queries**:
   - **Text Query**: "sunset over mountains"  
     Output: An image of a sunset over mountains along with descriptive text
   - **PDF Upload**: Upload a PDF of a scientific paper  
     Output: Extracted key sections or contextually relevant images

---

## ⚙️ Configuration
- 📊 **Vector Database**: It uses FAISS for efficient similarity search
- 🤖 **Model**: Uses OpenAI CLIP for neural embedding generation
- ✍️ **Augmentation**: Optional LLM-based augmentation for text responses

---

## 🗺️ Roadmap
- [ ] Fine-tuning CLIP for domain-specific datasets
- [ ] Adding support for audio and video modalities
- [ ] Improving the re-ranking system for better contextual relevance
- [ ] Enhanced PDF parsing with semantic section segmentation

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests or bug fixes.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
## 🙏 Acknowledgments

- [OpenAI CLIP](https://openai.com/research/clip)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face](https://huggingface.co/)
