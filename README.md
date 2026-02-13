# Build Your Own AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> *"What I cannot create, I do not understand."* — Richard Feynman

**A curated collection of tutorials for building modern AI systems from scratch.**

Master AI/ML by recreating the technologies behind ChatGPT, Stable Diffusion, AI Agents, and more — from an empty file to a working system. Inspired by [build-your-own-x](https://github.com/codecrafters-io/build-your-own-x).

---

## Get Started

If you're new here, this is the recommended learning path:

1. **Foundations** — Start with [micrograd](https://github.com/karpathy/micrograd) to understand neural networks and backpropagation from scratch
2. **Transformers** — Work through [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) to understand the architecture behind modern AI
3. **Build a GPT** — Follow [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy to build a language model end-to-end
4. **Go deeper** — Pick any category below that interests you and build!

---

## Table of Contents

- [Build your own `ChatGPT / LLM`](#build-your-own-chatgpt--llm)
- [Build your own `Image Generator`](#build-your-own-image-generator)
- [Build your own `Voice Assistant`](#build-your-own-voice-assistant)
- [Build your own `Code Copilot`](#build-your-own-code-copilot)
- [Build your own `AI Agent`](#build-your-own-ai-agent)
- [Build your own `RAG System`](#build-your-own-rag-system)
- [Build your own `Search Engine`](#build-your-own-search-engine)
- [Build your own `Recommendation Engine`](#build-your-own-recommendation-engine)
- [Build your own `Object Detector`](#build-your-own-object-detector)
- [Build your own `Neural Network`](#build-your-own-neural-network)
- [Build your own `Transformer`](#build-your-own-transformer)
- [Build your own `Fine-tuning / RLHF Pipeline`](#build-your-own-fine-tuning--rlhf-pipeline)
- [Build your own `Multimodal Model`](#build-your-own-multimodal-model)
- [Build your own `Text-to-Speech`](#build-your-own-text-to-speech)
- [Build your own `Embedding Model`](#build-your-own-embedding-model)
- [Build your own `AI Tool Use / Function Calling`](#build-your-own-ai-tool-use--function-calling)
- [Build your own `Autonomous Agent (RL)`](#build-your-own-autonomous-agent-rl)
- [Build your own `Translator`](#build-your-own-translator)
- [Build your own `OCR`](#build-your-own-ocr)

---

## Tutorials

### Build your own `ChatGPT / LLM`

* [**Build a Large Language Model (From Scratch)**](https://github.com/rasbt/LLMs-from-scratch) - Implementing a ChatGPT-like LLM from scratch, step by step, based on Sebastian Raschka's book [Python / PyTorch]
* [**Let's build GPT: from scratch, in code, spelled out**](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy's famous 2-hour tutorial building a Transformer following "Attention Is All You Need" [Python / PyTorch]
* [**Let's reproduce GPT-2 (124M)**](https://www.youtube.com/watch?v=l8pRSuU81PU) - Andrej Karpathy's 4-hour deep dive reproducing GPT-2 from an empty file [Python / PyTorch]
* [**nanoGPT**](https://github.com/karpathy/nanoGPT) - The simplest, fastest repository for training/finetuning medium-sized GPTs (~300 lines) [Python / PyTorch]
* [**build-nanogpt**](https://github.com/karpathy/build-nanogpt) - Video + code lecture on building nanoGPT from scratch with step-by-step git commits [Python / PyTorch]
* [**llm.c**](https://github.com/karpathy/llm.c) - LLM training in simple, raw C/CUDA — GPT-2 training without PyTorch or any framework [C / CUDA]
* [**picoGPT**](https://github.com/jaymody/picoGPT) - GPT-2 implemented in just 60 lines of NumPy, loads real GPT-2 weights [Python / NumPy]
* [**minGPT**](https://github.com/karpathy/minGPT) - A minimal PyTorch re-implementation of the OpenAI GPT training [Python / PyTorch]
* [**LLM101n**](https://github.com/karpathy/LLM101n) - Let's build a Storyteller — Karpathy's course building everything end-to-end from basics to a ChatGPT-like web app [Python / C / CUDA]
* [**nanochat**](https://github.com/karpathy/nanochat) - The best ChatGPT that $100 can buy — end-to-end pipeline to train a ChatGPT-style LLM from scratch [Python / PyTorch]
* [**Building LLaMA 3 from Scratch**](https://github.com/FareedKhan-dev/Building-llama3-from-scratch) - Recreating LLaMA 3's architecture in a simpler manner for learning purposes [Python / PyTorch]
* [**GPT in 60 Lines of NumPy**](https://jaykmody.com/blog/gpt-from-scratch/) - Blog post walking through the picoGPT implementation, explaining every line [Python / NumPy]

### Build your own `Image Generator`

* [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch) - Most-starred PyTorch implementation of DDPM (Ho et al., 2020) with clean, modular code [Python / PyTorch]
* [**The Annotated Diffusion Model**](https://huggingface.co/blog/annotated-diffusion) - Hugging Face's in-depth, line-by-line walkthrough of DDPM with math and code side-by-side [Python / PyTorch]
* [**Stable Diffusion from Scratch**](https://github.com/hkproj/pytorch-stable-diffusion) - Umar Jamil's complete Stable Diffusion using only PyTorch — covers text-to-image, image-to-image, and inpainting [Python / PyTorch]
* [**How Diffusion Models Work**](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) - DeepLearning.AI's free short course building a diffusion model from scratch with hands-on labs [Python / PyTorch]
* [**Hugging Face Diffusion Models Course**](https://github.com/huggingface/diffusion-models-class) - Free multi-unit course building diffusion models from scratch with Colab notebooks [Python / PyTorch]
* [**minDiffusion**](https://github.com/cloneofsimo/minDiffusion) - Self-contained, minimalistic implementation of diffusion models designed to be readable in a single sitting [Python / PyTorch]
* [**Conditional Diffusion MNIST**](https://github.com/TeaPearce/Conditional_Diffusion_MNIST) - Minimal single-script conditional diffusion model with Classifier-Free Guidance on MNIST [Python / PyTorch]
* [**Diffusion Models from Scratch**](https://chenyang.co/diffusion.html) - From-first-principles blog post combining rigorous math with working code [Python / PyTorch]
* [**How Diffusion Models Work — The Math from Scratch**](https://theaisummer.com/diffusion-models/) - Thorough blog post covering complete mathematical foundations with Python code [Python / PyTorch]
* [**labml.ai Stable Diffusion**](https://nn.labml.ai/diffusion/stable_diffusion/index.html) - Fully annotated implementation with side-by-side code and explanation of every component [Python / PyTorch]
* [**PyTorch-GAN**](https://github.com/eriklindernoren/PyTorch-GAN) - Collection of GAN implementations: Vanilla GAN, DCGAN, WGAN, CycleGAN, Pix2Pix, StyleGAN, and more [Python / PyTorch]
* [**pytorch-GANs**](https://github.com/gordicaleksa/pytorch-GANs) - Beginner-friendly GAN implementations with clear training scripts and visualizations [Python / PyTorch]

### Build your own `Voice Assistant`

* [**Whisper**](https://github.com/openai/whisper) - OpenAI's open-source speech recognition model — study the architecture to understand modern ASR [Python / PyTorch]
* [**Build Your Own Voice Assistant (Whisper + Ollama + Bark)**](https://medium.com/@vndee.huynh/build-your-own-voice-assistant-and-run-it-locally-whisper-ollama-bark-c80e6f815cba) - End-to-end local voice assistant combining STT, LLM, and TTS [Python / Whisper / Bark / Ollama]
* [**Build Your Own Private Voice Assistant (freeCodeCamp)**](https://www.freecodecamp.org/news/private-voice-assistant-using-open-source-tools/) - Privacy-first voice assistant using entirely open-source tools [Python]
* [**Real-Time Voice Cloning**](https://github.com/CorentinJ/Real-Time-Voice-Cloning) - Clone a voice in 5 seconds — three-stage pipeline: speaker encoder, synthesizer, vocoder [Python / PyTorch]
* [**Speech Recognition in Python**](https://thepythoncode.com/article/speech-recognition-in-python) - Comprehensive tutorial on building speech recognition systems from scratch [Python]

### Build your own `Code Copilot`

* [**Build a Coding Agent from Scratch**](https://sidbharath.com/blog/build-a-coding-agent-python-tutorial/) - Complete Python tutorial building a code-writing agent (~300 lines) step by step [Python]
* [**Training a Code-Generating LLM from Scratch**](https://medium.com/@wasiullah01/training-a-code-generating-llm-from-scratch-aff11c135733) - Specializing GPT-2 on GitHub code to understand code patterns and API usage [Python / PyTorch / HuggingFace]
* [**tabby**](https://github.com/TabbyML/tabby) - Self-hosted AI coding assistant — study the architecture of a code completion system [Rust / Python]
* [**Continue**](https://github.com/continuedev/continue) - Open-source AI code assistant — study how an IDE copilot is built [TypeScript / Python]
* [**Build an AI Agent in Python**](https://www.boot.dev/courses/build-ai-agent-python) - Boot.dev course building an AI agent that can write and execute code [Python]

### Build your own `AI Agent`

* [**AI Agent from Scratch in Python**](https://www.leoniemonigatti.com/blog/ai-agent-from-scratch-in-python.html) - Build an AI agent using LLM API calls, function calling, and conversation memory — no frameworks [Python]
* [**ReAct Agent from Scratch**](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/) - LangGraph's official tutorial on creating a custom ReAct agent with full control [Python / LangGraph]
* [**Simple ReAct Agent Implementation**](https://github.com/mattambrogi/agent-implementation) - Basic ReAct agent implementation in Python from scratch [Python]
* [**Multi-Agent System Design Patterns**](https://medium.com/aimonks/multi-agent-system-design-patterns-from-scratch-in-python-react-agents-e4480d099f38) - Building multi-agent systems from scratch with sequential and parallel patterns [Python]
* [**Build an AI Agent from Scratch (No Frameworks)**](https://alejandro-ao.com/agents-from-scratch/) - Pure Python and API calls, no LangChain or LlamaIndex [Python]
* [**Building ReAct Agents from Scratch with Gemini**](https://github.com/arunpshankar/react-from-scratch) - Collection of ReAct pattern examples optimized for Gemini [Python]
* [**AI Agents from Scratch: Single Agents**](https://towardsdatascience.com/ai-agents-from-zero-to-hero-part-1/) - Towards Data Science series on building agents from the ground up [Python]
* [**AI Agents for Beginners (Microsoft)**](https://github.com/microsoft/ai-agents-for-beginners) - Microsoft's 12-lesson open-source curriculum from concept to code [Python / Multi-framework]
* [**AutoGPT**](https://github.com/Significant-Gravitas/AutoGPT) - Study the architecture of an autonomous AI agent that breaks down complex objectives into subtasks [Python]

### Build your own `RAG System`

* [**RAG from Scratch — Learn by Building**](https://learnbybuilding.ai/tutorial/rag-from-scratch/) - Beginner's guide to building a RAG application from scratch [Python]
* [**Learn RAG from Scratch (freeCodeCamp)**](https://www.freecodecamp.org/news/mastering-rag-from-scratch) - In-depth course by a LangChain engineer covering RAG fundamentals [Python]
* [**Code a Simple RAG from Scratch**](https://huggingface.co/blog/ngxson/make-your-own-rag) - Hugging Face blog building a simple RAG system using Python and Ollama [Python]
* [**simple-local-rag**](https://github.com/mrdbourke/simple-local-rag) - Build a RAG pipeline from scratch and have it all run locally [Python]
* [**LangChain RAG Tutorial**](https://python.langchain.com/docs/tutorials/rag/) - Official LangChain tutorial building a RAG app in ~40 lines of code [Python / LangChain]
* [**Building RAG from Scratch (LlamaIndex)**](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/) - Low-level RAG implementation using open-source components [Python / LlamaIndex]
* [**Foundations of RAG Systems**](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/) - Crash course on building RAG systems with implementations [Python]
* [**7 Steps to Build a Simple RAG System from Scratch**](https://www.kdnuggets.com/7-steps-to-build-a-simple-rag-system-from-scratch) - KDnuggets step-by-step guide [Python]

### Build your own `Search Engine`

* [**Build a Vector Database from Scratch**](https://www.intoai.pub/p/build-a-vector-database-from-scratch) - Understanding RAG in depth by building a vector database [Python]
* [**Building a Vector Database from Scratch in Python**](https://osamadev.medium.com/building-a-vector-database-from-scratch-in-python-eb81c56e03fb) - Full implementation of a vector database in pure Python [Python]
* [**Build a Vector Search Engine with FAISS**](https://hackernoon.com/build-a-vector-search-engine-in-python-with-faiss-and-sentence-transformers) - Production-grade vector search from embedding generation through FAISS indexing to retrieval [Python / FAISS]
* [**Introduction to FAISS (Pinecone)**](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/) - Deep tutorial series covering IndexFlatL2, IVF, product quantization, and GPU acceleration [Python / FAISS]
* [**Faiss**](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search — study the source for efficient vector search algorithms [C++ / Python]
* [**txtai**](https://github.com/neuml/txtai) - All-in-one embeddings database for semantic search — study how semantic search is built [Python]
* [**Building a Semantic Search Engine**](https://huggingface.co/learn/nlp-course/chapter5/6) - Hugging Face NLP Course chapter on building semantic search with embeddings [Python]
* [**Qdrant**](https://github.com/qdrant/qdrant) - High-performance vector similarity search engine — learn vector DB internals from the source [Rust]

### Build your own `Recommendation Engine`

* [**Build a Recommendation Engine with Collaborative Filtering (Real Python)**](https://realpython.com/build-recommendation-engine-collaborative-filtering/) - User-based and item-based collaborative filtering, matrix factorization with SVD [Python / NumPy / Surprise]
* [**Recommender Systems in Python (DataCamp)**](https://www.datacamp.com/tutorial/recommender-systems-python) - Content-based and collaborative filtering recommendation engines [Python]
* [**Comprehensive Guide to Recommendation Engine**](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/) - Build a recommendation engine from scratch in Python [Python]
* [**Neural Collaborative Filtering with PyTorch**](https://www.marktechpost.com/2025/04/11/step-by-step-coding-guide-to-build-a-neural-collaborative-filtering-ncf-recommendation-system-with-pytorch/) - Implementing the NCF paper from scratch combining GMF and MLP [Python / PyTorch]
* [**Surprise**](https://github.com/NicolasHug/Surprise) - Python scikit for recommendation systems — study collaborative filtering algorithms [Python]
* [**LightFM**](https://github.com/lyst/lightfm) - Hybrid recommendation algorithm implementation — learn how to combine content and collaborative approaches [Python]
* [**Neural Collaborative Filtering**](https://github.com/hexiangnan/neural_collaborative_filtering) - Neural network-based collaborative filtering implementation [Python / Keras]

### Build your own `Object Detector`

* [**How to Implement YOLOv3 from Scratch (Paperspace, 5-Part Series)**](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) - The gold-standard tutorial: Darknet-53 backbone, detection layers, NMS, and full inference pipeline [Python / PyTorch]
* [**YOLO from Scratch**](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection) - Implementing YOLOv1, YOLOv3 from scratch in PyTorch with full training pipeline [Python / PyTorch]
* [**Training an Object Detector from Scratch**](https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/) - PyImageSearch tutorial building a custom object detector in PyTorch [Python / PyTorch]
* [**Understanding and Building Object Detection from Scratch**](https://www.analyticsvidhya.com/blog/2018/06/understanding-building-object-detection-model-python/) - Covering IOU, region-based approaches, and building detectors [Python]
* [**Building Object Detector from Scratch with TensorFlow**](https://medium.com/mlearning-ai/building-your-own-object-detector-from-scratch-with-tensorflow-bfeadfaddad8) - Deep learning object detector using plain Python, OpenCV, and TensorFlow [Python / TensorFlow]
* [**ultralytics**](https://github.com/ultralytics/ultralytics) - YOLOv8/v11 — study the state-of-the-art object detection implementation [Python / PyTorch]

### Build your own `Neural Network`

* [**micrograd**](https://github.com/karpathy/micrograd) - Andrej Karpathy's tiny autograd engine and neural net library — the best starting point [Python]
* [**Neural Networks: Zero to Hero**](https://karpathy.ai/zero-to-hero.html) - Karpathy's full video course building neural networks from scratch, from backprop to GPT [Python / PyTorch]
* [**Neural Networks from Scratch in Python**](https://nnfs.io/) - Harrison Kinsley's book and YouTube series building neural nets in pure Python, then NumPy [Python / NumPy]
* [**Neural Networks from Scratch (nnfs_book)**](https://github.com/Sentdex/nnfs_book) - Sample code from the NNFS book, separated by chapter [Python]
* [**tinygrad**](https://github.com/tinygrad/tinygrad) - A tiny deep learning framework in ~5000 lines — understand how autograd and tensor ops work [Python]
* [**3Blue1Brown Neural Networks**](https://www.3blue1brown.com/topics/neural-networks) - Visual explanations of neural networks, backpropagation, and gradient descent [Visual / Math]
* [**Neural Network from Scratch in Python**](https://victorzhou.com/blog/intro-to-neural-networks/) - Building a complete neural network from scratch with only NumPy [Python / NumPy]

### Build your own `Transformer`

* [**The Annotated Transformer**](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP's line-by-line implementation of "Attention Is All You Need" [Python / PyTorch]
* [**Transformer from Scratch using PyTorch**](https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb) - Complete implementation with multi-head attention, positional encoding, and training [Python / PyTorch]
* [**Building a Transformer with PyTorch (DataCamp)**](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch) - Step-by-step tutorial from theory to code [Python / PyTorch]
* [**Vision Transformer (ViT) from Scratch**](https://github.com/tintn/vision-transformer-from-scratch) - Simplified PyTorch implementation of ViT for image classification [Python / PyTorch]
* [**vit-pytorch**](https://github.com/lucidrains/vit-pytorch) - Implementation of Vision Transformer achieving SOTA in vision classification [Python / PyTorch]
* [**Build Your Own ViT Model from Scratch (freeCodeCamp)**](https://www.freecodecamp.org/news/build-your-own-vit-model-from-scratch/) - Comprehensive course building a Vision Transformer with PyTorch [Python / PyTorch]
* [**Understanding and Coding Self-Attention**](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) - Sebastian Raschka's guide to self-attention, multi-head attention, causal and cross-attention in LLMs [Python / PyTorch]
* [**The Attention Mechanism from Scratch**](https://machinelearningmastery.com/the-attention-mechanism-from-scratch/) - Implementing the general attention mechanism with NumPy [Python / NumPy]
* [**Building Transformer Models from Scratch (10-Day Mini-Course)**](https://machinelearningmastery.com/building-transformer-models-from-scratch-with-pytorch-10-day-mini-course/) - Each lesson covers a specific transformer component with PyTorch implementation [Python / PyTorch]

### Build your own `Fine-tuning / RLHF Pipeline`

* [**LLMs-from-scratch: DPO from Scratch**](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb) - Sebastian Raschka's notebook implementing Direct Preference Optimization from scratch and applying it to a GPT-like LLM [Python / PyTorch]
* [**Reinforcement Learning in LLM Tutorial (from scratch)**](https://github.com/tsmatz/reinforcement-learning-in-llm) - Step-by-step notebooks covering RLHF with PPO, DPO, and GRPO from scratch in pure Python [Python / PyTorch]
* [**Code LoRA from Scratch (Lightning AI)**](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch) - Sebastian Raschka's interactive studio explaining LoRA by coding it from scratch in PyTorch [Python / PyTorch]
* [**LoRA from Scratch**](https://github.com/sunildkumar/lora_from_scratch) - Implements Low-Rank Adaptation finetuning from scratch, guided step-by-step by the original LoRA paper [Python / PyTorch]
* [**Direct Preference Optimization from Scratch in PyTorch**](https://github.com/0xallam/Direct-Preference-Optimization) - Clean PyTorch implementation of DPO replacing RLHF's reward model with a simple preference loss [Python / PyTorch]
* [**The N Implementation Details of RLHF with PPO**](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) - Hugging Face deep dive into OpenAI's RLHF codebase with a minimal reference implementation [Python / PyTorch]
* [**LLM-RLHF-Tuning-with-PPO-and-DPO**](https://github.com/raghavc/LLM-RLHF-Tuning-with-PPO-and-DPO) - Comprehensive toolkit covering instruction fine-tuning, reward model training, PPO and DPO for LLaMA models [Python / PyTorch]
* [**OpenRLHF**](https://github.com/OpenRLHF/OpenRLHF) - Scalable RLHF framework built on Ray and vLLM with clean implementations of PPO, GRPO, and REINFORCE++ [Python / PyTorch / Ray]

### Build your own `Multimodal Model`

* [**nanoVLM**](https://github.com/huggingface/nanoVLM) - Hugging Face's simplest repository to train a Vision-Language Model from scratch in ~750 lines of pure PyTorch [Python / PyTorch]
* [**seemore: Vision Language Model from Scratch**](https://github.com/AviSoori1x/seemore) - From-scratch VLM in pure PyTorch — vision encoder, projector, and language decoder all built from the ground up [Python / PyTorch]
* [**Coding a Multimodal (Vision) Language Model from Scratch**](https://www.youtube.com/watch?v=vAmKB7iPkWw) - Umar Jamil's video coding PaliGemma from scratch in PyTorch with full explanation [Python / PyTorch]
* [**Building CLIP from Scratch**](https://github.com/moein-shariatnia/OpenAI-CLIP) - Simple, educational PyTorch implementation of CLIP with contrastive learning on image-text pairs [Python / PyTorch]
* [**simple-clip**](https://github.com/filipbasara0/simple-clip) - Minimal but effective CLIP implementation designed for readability and learning [Python / PyTorch]
* [**Training a CLIP Model from Scratch**](https://learnopencv.com/clip-model/) - LearnOpenCV tutorial training CLIP end-to-end with contrastive loss and evaluation [Python / PyTorch]
* [**Understanding Multimodal LLMs**](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) - Sebastian Raschka's deep dive into how vision-language models work with architectural explanations [Article]

### Build your own `Text-to-Speech`

* [**Build Text-to-Speech from Scratch**](https://medium.com/@tttzof351/build-text-to-speech-from-scratch-part-1-ba8b313a504f) - Multi-part tutorial building a transformer-based TTS model from scratch using LJSpeech [Python / PyTorch]
* [**NVIDIA Tacotron 2**](https://github.com/NVIDIA/tacotron2) - NVIDIA's PyTorch implementation of Tacotron 2 with faster-than-realtime inference [Python / PyTorch]
* [**VITS**](https://github.com/jaywalnut310/vits) - Official implementation of end-to-end TTS combining variational inference, normalizing flows, and adversarial training [Python / PyTorch]
* [**VITS Training Tutorial**](https://github.com/Pipe1213/VITS_Tutorial) - Step-by-step guide on training VITS from scratch using your own dataset [Python / PyTorch]
* [**Coqui TTS**](https://github.com/coqui-ai/TTS) - Battle-tested deep learning TTS toolkit supporting Tacotron, Glow-TTS, VITS and multiple vocoders [Python / PyTorch]
* [**Grad-TTS**](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) - Official implementation of diffusion-based TTS that gradually transforms noise into mel-spectrograms [Python / PyTorch]
* [**Text-to-Speech with Tacotron2 (PyTorch Tutorial)**](https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html) - Official torchaudio tutorial building the full TTS pipeline: text to spectrogram to waveform [Python / PyTorch]
* [**WaveNet Vocoder**](https://github.com/r9y9/wavenet_vocoder) - PyTorch implementation of WaveNet vocoder for understanding neural audio synthesis from scratch [Python / PyTorch]

### Build your own `Embedding Model`

* [**SimCSE**](https://github.com/princeton-nlp/SimCSE) - Princeton NLP's official implementation — unsupervised and supervised contrastive sentence embeddings from scratch [Python / PyTorch]
* [**simple-simcse**](https://github.com/hppRC/simple-simcse) - Minimal SimCSE implementation with almost no external dependencies, built for learning the internals [Python / PyTorch]
* [**Training Embedding Models with Sentence Transformers v3**](https://huggingface.co/blog/train-sentence-transformers) - Hugging Face's definitive guide to training custom embedding models with losses, evaluators, and datasets [Python / Sentence Transformers]
* [**Word2Vec from Scratch**](https://jaketae.github.io/study/word2vec/) - Implementing Skip-gram and CBOW word embeddings from scratch using only NumPy [Python / NumPy]
* [**Matryoshka Representation Learning from the Ground Up**](https://aniketrege.github.io/blog/2024/mrl/) - Tutorial implementing Matryoshka embeddings that work at multiple dimensions [Python / PyTorch]
* [**Introduction to Matryoshka Embedding Models**](https://huggingface.co/blog/matryoshka) - Hugging Face guide to training Matryoshka models for flexible-dimension embeddings [Python / Sentence Transformers]

### Build your own `AI Tool Use / Function Calling`

* [**AI Agents from Scratch**](https://github.com/pguso/ai-agents-from-scratch) - Progressive lessons from tool calling to ReAct agents with local LLMs — no black boxes [Python]
* [**Function Calling Using LLMs (Martin Fowler)**](https://martinfowler.com/articles/function-call-LLM.html) - In-depth article explaining how LLMs interact with external tools, building a Shopping Agent step by step [Python]
* [**Create a ReAct AI Agent from Scratch**](https://shafiqulai.github.io/blogs/blog_3.html) - Build a complete ReAct agent with tool use from scratch in pure Python [Python]
* [**Integrating with Function Calling (Microsoft)**](https://github.com/microsoft/generative-ai-for-beginners/blob/main/11-integrating-with-function-calling/README.md) - Lesson 11 of Microsoft's 21-lesson course covering function calling from fundamentals to integration [Python]
* [**How to Build a Custom MCP Server with TypeScript**](https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/) - freeCodeCamp handbook building an MCP server from scratch [TypeScript / MCP]
* [**Build an MCP Server (Official Tutorial)**](https://modelcontextprotocol.io/docs/develop/build-server) - Official Model Context Protocol documentation for building MCP servers from scratch [Python / TypeScript / MCP]

### Build your own `Autonomous Agent (RL)`

* [**Spinning Up in Deep RL**](https://github.com/openai/spinningup) - OpenAI's educational resource with clean implementations of VPG, TRPO, PPO, DDPG, TD3, SAC [Python / PyTorch]
* [**Reinforcement Learning (DQN) Tutorial**](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) - Official PyTorch tutorial training a DQN agent on CartPole [Python / PyTorch]
* [**Reinforcement Learning (PPO) with TorchRL**](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html) - Official PyTorch tutorial implementing PPO for the Inverted Pendulum [Python / PyTorch]
* [**Q-Learning from Scratch in Python**](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/) - Tabular Q-learning from scratch, showing every step from state space design to convergence [Python / Gymnasium]
* [**Teach AI to Play Snake — RL with PyTorch**](https://www.python-engineer.com/posts/teach-ai-snake-reinforcement-learning/) - 4-part series building a DQN agent that learns to play Snake with Pygame [Python / PyTorch / Pygame]
* [**CleanRL**](https://github.com/vwxyzjn/cleanrl) - Single-file implementations of RL algorithms with tracked experiments [Python / PyTorch]
* [**Policy Gradient with PyTorch**](https://huggingface.co/blog/deep-rl-pg) - Hugging Face Deep RL Course chapter implementing REINFORCE from scratch [Python / PyTorch]
* [**Deep Reinforcement Learning Course**](https://huggingface.co/learn/deep-rl-course/unit0/introduction) - Hugging Face's free course from fundamentals to advanced RL [Python / PyTorch]
* [**Gymnasium**](https://github.com/Farama-Foundation/Gymnasium) - Standard API for RL environments — build and test your own agents [Python]

### Build your own `Translator`

* [**NLP From Scratch: Translation with Seq2Seq and Attention**](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) - Official PyTorch tutorial implementing encoder-decoder with Bahdanau attention for translation [Python / PyTorch]
* [**Sequence to Sequence Learning with Neural Networks**](https://github.com/bentrevett/pytorch-seq2seq) - 6 notebooks progressing from basic seq2seq to full Transformer for translation [Python / PyTorch]
* [**Neural Machine Translation with Transformer (TensorFlow)**](https://www.tensorflow.org/text/tutorials/transformer) - Full Transformer from scratch for Portuguese-to-English translation [Python / TensorFlow]
* [**The Annotated Transformer**](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP's annotated implementation — the original Transformer was built for translation [Python / PyTorch]
* [**Language Translation with nn.Transformer**](https://pytorch.org/tutorials/beginner/translation_transformer.html) - Official PyTorch tutorial building a translation model with Transformer [Python / PyTorch]
* [**Helsinki-NLP/Opus-MT**](https://github.com/Helsinki-NLP/Opus-MT) - Open neural machine translation models — study how production translation systems work [Python]
* [**fairseq**](https://github.com/facebookresearch/fairseq) - Facebook AI Research's sequence-to-sequence toolkit with translation models [Python / PyTorch]

### Build your own `OCR`

* [**CRNN + CTC for OCR**](https://github.com/Holmeyoung/crnn-pytorch) - PyTorch implementation of CRNN with CTC loss for all-language OCR [Python / PyTorch]
* [**PyTorch CRNN: Seq2Seq Digits Recognition with CTC**](https://codingvision.net/pytorch-crnn-seq2seq-digits-recognition-ctc) - Step-by-step tutorial building a CRNN for sequence recognition [Python / PyTorch]
* [**Handwriting Recognition with PyTorch**](https://pylessons.com/handwriting-recognition-pytorch) - Building a handwriting recognition system from scratch [Python / PyTorch]
* [**EasyOCR**](https://github.com/JaidedAI/EasyOCR) - Ready-to-use OCR supporting 80+ languages — study the architecture of a production OCR system [Python / PyTorch]
* [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) - Multilingual OCR toolkit — study text detection, recognition, and layout analysis pipelines [Python / PaddlePaddle]
* [**Building a Handwriting Recognition System with CRNN**](https://medium.com/@pavitharan2020/building-a-handwriting-recognition-system-with-crnn-a-beginners-guide-58a51a46dd15) - Beginner's guide explaining CNN + RNN + CTC architecture [Python / PyTorch]

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding tutorials.

If you find this list useful, please give it a star. It helps others discover it too.

## License

[MIT](LICENSE)
