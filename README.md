# Assignment 1
In this assignment, regularized linear and logistic regression were implemented from scratch using PyTorch’s torch.xyz namespace, avoiding derived libraries. For linear regression, both Stochastic Gradient Descent (SGD) and SGD with momentum were developed, with loss vs. epoch plots comparing their convergence speeds. In logistic regression, the Click-Through Rate (CTR) dataset was preprocessed, and a model was trained using SGD. All gradient calculations and weight updates were explicitly derived and formatted in LaTeX. Hyperparameters were carefully chosen, and performance was analyzed using a precision-recall curve, demonstrating the trade-offs between precision and recall in classification.

# Assignment 2a
In this assignment, explainer algorithms were implemented to analyze a CNN model trained on the cats vs. dogs classification task. The model, benefiting from data augmentation, achieved high accuracy without overfitting. To interpret its decision-making process, Integrated Gradients and Grad-CAM were implemented using PyTorch and Captum. The notebook included detailed markdown explanations in a tutorial format, ensuring accessibility for those familiar with CNNs. Results from both methods provided visual insights into model predictions, highlighting important regions in the images. The implementation helped in understanding the model’s learned features and decision boundaries.

# Assignment 2b
In this assignment, anomaly detection was performed on the MVTec-AD dataset using PatchCore and EfficientAD models via the anomalib library. Only flat-surface categories (tile, leather, grid) were analyzed. AUROC scores were reported at the product category level and as an overall average. A tutorial-style report explained the methods, results, and concepts like coresets. Additionally, feature extraction was conducted to perform similarity search on anomalous cases using the Qdrant vector database. The system accepted an input image and retrieved the top 5 similar images, ensuring anomalous cases were matched to similar anomalies.

# Assignment 3
In this assignment, object detection and tracking were performed using YOLO, DeepSORT, OpenCV, and Torch. The notebook installed Torch, TorchVision, OpenCV, and yt-dlp for preprocessing and model execution. A YouTube video was downloaded at the highest resolution using yt-dlp, and frames were processed using OpenCV. YOLO was used for object detection, while DeepSORT handled multi-object tracking. The implementation involved frame-by-frame analysis, real-time tracking, and video annotation. The results were visualized, demonstrating the effectiveness of object detection and tracking algorithms in dynamic environments.

# Project
This project involved developing a Retrieval-Augmented Generation (RAG) system to assist ROS2 robotics developers in building navigation stacks for agents with egomotion. The system focused on answering domain-specific questions across ROS2 middleware, Nav2 navigation, MoveIt2 motion planning, and Gazebo simulation subdomains.

## Project Milestones
### Environment and Tooling:
A Docker Compose setup was created, orchestrating services like MongoDB (data storage), Qdrant (vector search), and ClearML (experiment tracking). The system enabled model training and Hugging Face Hub API interactions. Screenshots validated deployment success.
### ETL Pipeline:
The ClearML orchestrator was used to build an ETL pipeline, extracting structured data from ROS2 documentation and YouTube transcripts. The pipeline stored raw data in MongoDB, and URLs were logged via queries.
### Featurization Pipeline:
The raw data was processed into vector embeddings and stored in MongoDB and Qdrant, enabling efficient similarity-based retrieval.
### Fine-tuning with LoRA:
Fine-tuning was performed using LoRA (Low-Rank Adaptation), a parameter-efficient adaptation method for large language models. Instead of modifying all model weights, LoRA injected trainable low-rank matrices into transformer layers, significantly reducing computational overhead while maintaining accuracy. This approach enabled fine-tuning on navigation-specific queries without requiring extensive GPU resources.
### Prompt Engineering:
Carefully designed prompts were developed to optimize response relevance and coherence. Techniques such as few-shot prompting, chain-of-thought reasoning, and instruction tuning were used to guide the model in generating accurate, structured answers for ROS2-related queries.
### Deployment:
A Gradio app was developed, integrating Ollama for inference and allowing direct retrieval from Hugging Face Hub. Predefined questions like "How can I navigate to a specific pose?" were selectable, generating structured and actionable responses.

This project successfully demonstrated RAG’s capability in enhancing domain-specific robotics problem-solving through efficient fine-tuning and retrieval techniques.
