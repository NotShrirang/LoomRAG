# 🌟 LoomRAG: Multimodal Retrieval-Augmented Generation for AI-Powered Search

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system, named **LoomRAG**, that leverages OpenAI's CLIP model for neural cross-modal retrieval and semantic search. The system allows users to input text queries and retrieve both text and image responses seamlessly through vector embeddings. It also supports uploading images and PDFs for enhanced interaction and intelligent retrieval capabilities through a Streamlit-based interface.

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
